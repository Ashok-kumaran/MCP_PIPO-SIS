import os
import boto3
from dotenv import load_dotenv

load_dotenv()

class ObjectStoreClient:
    def __init__(self):
        self.bucket = os.getenv("BUCKET_NAME")
        self.region = os.getenv("REGION")
        self.endpoint_url = os.getenv("ENDPOINT_URL")

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("READ_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("READ_SECRET_ACCESS_KEY"),
            region_name=self.region,
            endpoint_url=self.endpoint_url,
        )

    def read_object(self, key: str) -> str:
        """
        Reads an object from the bucket and returns content as string.
        """
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=key
        )
        return response["Body"].read().decode("utf-8")
