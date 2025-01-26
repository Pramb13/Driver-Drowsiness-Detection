from setuptools import setup, find_packages

setup(
    name="drowsiness-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "dlib",
        "numpy",
        "scipy",
        "playsound",
    ],
    entry_points={
        "console_scripts": [
            "drowsiness-detection = app:main",  # Update with the correct entry point
        ]
    },
    include_package_data=True,
    package_data={
        "": ["models/*.dat", "alarm.wav"],  # Make sure these files are included
    }
)
