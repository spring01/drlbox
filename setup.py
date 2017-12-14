
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='DRLBOX',
    version='0.0.1',
    description="Deep Reinforcement Learning as a (black) BOX",
    long_description=long_description,
    url='https://github.com/spring01/drlbox',
    author='Haichen Li',
    author_email='lihc2012@gmail.com',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='deep reinforcement learning',
    packages=find_packages(),
    install_requires=['tensorflow>=1.3', 'gym>=0.9.3'],
    scripts=['bin/drlbox_a3c.py',
             'bin/drlbox_dqn.py',
             'bin/drlbox_evaluator.py'],
)
