from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mebm',
    version='0.0.1',
    description='Moist energy balance model',
    long_description=readme,
    author='Henry Peterson',
    author_email='hankpete97@gmail.com',
    url='https://github.com/hankpete/EBM',
    license=license,
    packages=['mebm'],
    include_package_data=True
)
