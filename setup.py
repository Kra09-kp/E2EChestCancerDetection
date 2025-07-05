import setuptools
from dotenv import load_dotenv
import os

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()



load_dotenv()

# Load environment variables
__version__  = os.getenv("VERSION")
SRC_REPO = os.getenv("SRC_REPO")
AUTHOR_USER_NAME = os.getenv("AUTHOR_USER_NAME")
AUTHOR_EMAIL = os.getenv("AUTHOR_EMAIL")
REPO_NAME = os.getenv("REPO_NAME")


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for Chest Cancer Detection using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages=setuptools.find_packages(where="src"),   
)