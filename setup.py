import setuptools
import os


SRC_REPO="cnnClassifier"
AUTHOR_USER_NAME = "Kra09-kp"
AUTHOR_EMAIL="kirtipogra@gmail.com"
REPO_NAME="E2EChestCancerDetection"
__version__ = "0.1.0"

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()

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