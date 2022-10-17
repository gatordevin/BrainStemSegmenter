from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from time import sleep
import re
import os
import shutil
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
        for file in file_list:
            if(file["mimeType"]=="application/vnd.google-apps.folder"):
                print("Folder found : " + str(file["title"]))
                folders.append(file)
            if("image" in file["mimeType"]):
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
                section_split = section.split("r")
                section_row = section_split[-1]
                section_column = section_split[-2].split("c")[-1]
                if(section not in filtered_images.keys()):
                    filtered_images[section] = []
                filtered_images[section].append({
                    "sub_image_number" : int(sub_image_number),
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

def download_images(filtered_dict):
    for key in filtered_dict.keys():
        os.makedirs("tmp_drive_images/"+key, exist_ok=True)
        for image_dict in filtered_dict[key]:
            image_dict["file"].GetContentFile("tmp_drive_images/"+key+"/"+image_dict["file"]["title"])
            
    
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
        download_images(image_dicts)
        shutil.rmtree("tmp_drive_images")
        # for image in image_ids[key]["images"]:
        #     print("     " + image["title"])
        # drive_file.Upload()
        # drive_file['parents'] = [{"kind": "drive#parentReference", "id": target_id}]
        # drive_file.Upload()
        # print("moving")

    sleep(1)
# fileList = drive.ListFile({'q': "'Fuller Lab Images' in parents"}).GetList()

# for file1 in fileList:
#     print('title: %s, id: %s' % (file1['title'], file1['id']))