from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from time import sleep
import matplotlib.pyplot as plt
import re
import os
import shutil
import cv2
from PIL import Image, ImageDraw
# Rename the downloaded JSON file to client_secrets.json
# The client_secrets.json file needs to be in the same directory as the script.
gauth = GoogleAuth()
drive = GoogleDrive(gauth)
# print(drive.LIS)
# List files in Google Drive
file_list = drive.ListFile({'q': "'root' in parents and trashed=false and mimeType = 'application/vnd.google-apps.folder'"}).GetList()

target_folder = "Fuller Lab Images"
target_id = ""
for drive_file in file_list:
    if drive_file["title"]==target_folder:
        target_id = drive_file["id"]
        break

def find_images(folder):
    found_images = {}
    print("looking for images")
    folders = [folder]
    while len(folders) > 0:
        file_list = drive.ListFile({'q': "'{0}' in parents and trashed=false".format(folders[0]["id"])}).GetList()
        folder = folders.pop(0)
        already_stitched = False
        for file in file_list:
            if("Stitched" in file["title"]):
                already_stitched = True 
                break
        if(already_stitched):
            continue
        for file in file_list:
            if(file["mimeType"]=="application/vnd.google-apps.folder"):
                print("Folder found : " + str(file["title"]))
                folders.append(file)
            if("image" in file["mimeType"]):
                # print(file)
                if folder["id"] not in found_images.keys():
                    found_images[folder["id"]] = {}
                    found_images[folder["id"]]["file"] = folder
                    found_images[folder["id"]]["images"] = []
                found_images[folder["id"]]["images"].append(file)
        print("Files left : " + str(len(folders)))
    return found_images

def filter_images(files):
    filtered_images = {}
    print("Filtering images")
    for file in files:
        file_name = file["title"]
        file_split = file_name.split(".")
        print("Analyzing : " + str(file_name))
        if len(file_split) >= 3:
            section = 0
            take_number = 0
            sub_image = 0
            extension = 0
            if(len(file_split)==3):
                section, sub_image, extension = file_name.split(".")
                take_number = 1
            elif(len(file_split)==4):
                section, take_number, sub_image, extension = file_name.split(".")
            if("tif" in extension):
                sub_image_number = re.sub("[^0-9]", "", sub_image)
                stain = sub_image[-1]
                section_split = section.split("r")
                section_row = section_split[-1]
                section_column = section_split[-2].split("c")[-1]
                image_group = section+stain
                if(image_group not in filtered_images.keys()):
                    filtered_images[image_group] = []
                filtered_images[image_group].append({
                    "sub_image_number" : int(sub_image_number),
                    "stain" : stain,
                    "section_row" : int(section_row),
                    "section_column" : int(section_column),
                    "take" : take_number,
                    "file" : file,
                    }
                )
                print("Valid Tiff section row : " + str(section_row) + " : section_column : " + str(section_column) + " : sub image number : " + str(sub_image_number) + " : take number : " + str(take_number))
        else:
            print("Invalid tif")
    for key in filtered_images.keys():
        filtered_images[key].sort(key=lambda x: x["sub_image_number"])
    return filtered_images

def download_and_stitch_images(filtered_dict, save_id, save_name):
        os.makedirs("tmp_drive_images/"+key, exist_ok=True)
        images = []
        stitcher = cv2.Stitcher_create()
        take_number = 0
        for image_dict in filtered_dict:
            if(int(image_dict["take"])>take_number):
                take_number = int(image_dict["take"])
        for image_dict in filtered_dict:
            if(int(image_dict["take"]) == take_number):
                image_dict["file"].GetContentFile("tmp_drive_images/"+key+"/"+image_dict["file"]["title"])
                image = cv2.imread("tmp_drive_images/"+key+"/"+image_dict["file"]["title"])
                images.append(image)
        print(len(images))
        (status, stitched) = stitcher.stitch(images)
        if(status==0):
            # plt.imshow(stitched*2)
            # plt.show()
            img = Image.fromarray(stitched, "RGB").convert("L")
            img.save("tmp_drive_images/"+ key + "/PM.tiff", 'TIFF')
            image_upload = drive.CreateFile({
                    'title': save_name + "_PM",
                    'parents': [{'id': save_id}],
                    'mimeType': 'image/tiff'
                    })
            image_upload.SetContentFile("tmp_drive_images/"+ key + "/PM.tiff")
            image_upload.Upload()
            del image_upload
        else:
            print("Stitch failed with error code: " + str(status) + " : " + key)
        del stitcher
        while(True):
            try:
                shutil.rmtree("tmp_drive_images/"+key)
                break
            except PermissionError:
                print("Image still be uploaded")
                sleep(1)
            
        
            
    
while(True):
    target_folder_file_list = drive.ListFile({'q': "'{0}' in parents and trashed=false".format(target_id)}).GetList()
    stitch_title = "Stitch"
    stitch_id = ""
    stitch_file = None
    for drive_file in target_folder_file_list:
        if(drive_file["title"]==stitch_title):
            stitch_id = drive_file["id"]
            stitch_file = drive_file
    
    assert stitch_id != ""
    image_ids = find_images(stitch_file)
    for key in image_ids.keys():
        print(image_ids[key]["file"]["title"])
        image_dicts = filter_images(image_ids[key]["images"])
        print("images_filtered")

        file_metadata = {
        'title': 'Stitched',
        'parents': [{'id': key}],
        'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = drive.CreateFile(file_metadata)
        folder.Upload()
        while(not folder.uploaded):
            print("Creating photomerge folder")
            sleep(1)
        print(folder["id"])

        for key in image_dicts.keys():
            image = download_and_stitch_images(image_dicts[key], folder["id"], key)
        # After downloading images need to have them ready and loaded into numpy array
        # Will then photomerge the image and upload to google drive
        # Photomerged image should go in the root animal folder with photomerged folder name
        # Not only does the tmp directory for stiching need to be deleted but the name of the root dir needs to be rememebred and not restiched
        # It also needs to be removed from the stitching folder
        # Once removed from the folder we can move it from the ban list in case it gets moved back in
        # This should make stitching as simple as dragging the images into stitch folder after upload then just waiting
        # Eventually they will stitch and get booted from that folder with the fully stitched image
        # Now for future folders that will have analysis and such it will by default find and used the photomerged images only

    sleep(1)
# fileList = drive.ListFile({'q': "'Fuller Lab Images' in parents"}).GetList()

# for file1 in fileList:
#     print('title: %s, id: %s' % (file1['title'], file1['id']))