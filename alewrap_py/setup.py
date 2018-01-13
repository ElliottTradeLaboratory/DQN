from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'alewrap_py/package_data.txt')) as f:
    package_data = [line.rstrip() for line in f.readlines()]

setup(
    name='alewrap_py',
    version='1.0.0',
    author="ElliottTradeLaboratory",
    author_email="elliott.trade.laboratory@gmail.com",
    description="alewrap for Python",
    license="GPL",
    url="https://github.com/ElliottTradeLaboratory/alewrap_py",
    install_requires=[
        'opencv-python>=3.3.0.10',
        'sk-video>=1.1.8'
    ],
    packages=['alewrap_py'],
    package_data={'alewrap_py': [
        'alewrap_py/*.py',
        'alewrap_py/xitari/libxitari.so',
    ]},
    cmdclass={'build': Build},
)
