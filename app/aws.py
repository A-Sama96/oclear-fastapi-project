import re
import boto3
from cloudpathlib import CloudPath
S3_BUCKET_NAME = "oclear-107898776944-bucket"
# S3_CHECK_URI = 's3://'+S3_BUCKET_NAME+'/ecobank-check-s3/'
# S3_SIGNATURE_URI = 's3://'+S3_BUCKET_NAME+'/ecobank-signature-s3/'
S3_CHECK_FOLDERNAME = 'ecobank-check-s3'
S3_SIGNATURE_FOLDERNAME = 'ecobank-signature-s3'

s3 = boto3.resource('s3')


def download_all_files_in_folder_from_s3(s3_bucket_name: str, s3_folder: str, destination_folder: str) -> None:
    list_s3_files = list_s3_files_in_folder(s3_bucket_name, s3_folder)
    for filename in list_s3_files:
        s3_file_uri = f"s3://{s3_bucket_name}/{s3_folder}/{filename}"
        download_file_from_s3(s3_file_uri, destination_folder)


def download_file_from_s3(s3_file_uri: str, destination_folder: str) -> None:
    cp = CloudPath(s3_file_uri)
    cp.download_to(destination_folder)


def list_s3_files_in_folder(s3_bucket_name: str, folder_name: str) -> list:
    """
    This functions list files in a folder from s3 bucket using s3 resource object.
    """
    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(s3_bucket_name)
    substring = folder_name
    list_files = []
    for obj in s3_bucket.objects.all():
        if re.search(substring,  obj.key):
            list_files.append(obj.key.split('/')[1])
    return list_files
    # files = s3_bucket.objects.all()
    # for file in files:
    #     print(file)


# print(list_s3_files_in_folder("oclear-107898776944-bucket", "ecobank-signature-s3"))
