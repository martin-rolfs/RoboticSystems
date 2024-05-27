from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Simple implementations of basic robot control functionalities.'
LONG_DESCRIPTION = 'Defines a robot model and a trajectory class that can be used to model and control a robot. Interfaces to simulators or physical robots are not implemented.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="robotic_systems", 
        version=VERSION,
        author="Martin Rolfs",
        author_email="<rolfs.martin@googlemail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'sympy', 'scipy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'robot'],
        classifiers= [
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)