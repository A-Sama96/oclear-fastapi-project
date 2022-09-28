import re
import boto3
from cloudpathlib import CloudPath

S3_CHECK_URI = 's3://oclear-107898776944-bucket/ecobank-check-s3/'
S3_SIGNATURE_URI = 's3://oclear-107898776944-bucket/ecobank-signature-s3/'

s3 = boto3.resource('s3')


def download_file_from_s3(s3_file_uri: str, destination_folder: str):
    cp = CloudPath(s3_file_uri)
    cp.download_to(destination_folder)


def list_s3_files_in_folder(s3_bucket_name: str, folder_name: str):
    """
    This functions list files from s3 bucket using s3 resource object.
    :return: None
    """
    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(s3_bucket_name)
    substring = folder_name

    for obj in s3_bucket.objects.all():
        if re.search(substring,  obj.key):
            print(obj.key)

    # files = s3_bucket.objects.all()
    # for file in files:
    #     print(file)
