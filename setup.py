from setuptools import setup, find_packages
setup(
    name='CosmiQ_SN4_Baseline',
    version='0.1.1',
    description='SpaceNet 4 Baseline',
    author='Nick Weir',
    author_email='nweir@iqt.org',
    license='APACHE 2',
    packages=find_packages(),
    include_package_data=True,

    dependency_links=['https://github.com/SpaceNetChallenge/utilities.git@spacenetV3#egg=spacenetutilites-3.0'],

    install_requires=['numpy>=1.15', 'pandas>=0.23.4', 'scikit-image>=0.14.0',
                      'rasterio>=1.0.7', 'opencv>=3.4.3',
                      'spacenetutilities==3.0', 'keras>=2.2.2',
                      'tensorflow>=1.10.0', 'tensorboard>=1.10.0',
                      'setuptools>=40.4.0'],  # TODO: UPDATE
    entry_points={
        'console_scripts': [
            'make_np_arrays = bin/make_np_arrays:main_func',
            'train_model = bin/train_model:main_func',
            'make_predictions = bin/make_predictions:main_func'
        ]
    },

    package_data={
        "": ['*.txt', '*.md', 'Dockerfile']
    },

    python_requires='>=3.6.5',


)
