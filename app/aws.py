import boto3

S3_CHECK_URI = 's3://oclear-107898776944-bucket/ecobank-check-s3/'
S3_SIGNATURE_URI = 's3://oclear-107898776944-bucket/ecobank-signature-s3/'

s3 = boto3.resource('s3')
