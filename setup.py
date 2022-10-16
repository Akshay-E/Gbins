import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='Gbins',
    version='0.0.1',
    author='Akshay-E',
    author_email='eakshay0109@gmail.com',
    description='Broadband injections in radio astronomical data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Akshay-E/Gbins.git',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
       'numpy',
       'scipy',
       'setigen',
       'matplotlib',
       'shutil',
       'warnings',
    ],

    
)