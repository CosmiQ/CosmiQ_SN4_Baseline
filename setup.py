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

    install_requires=['numpy', 'pandas', 'scikit-image',
                      'rasterio>=1.0.7', 'opencv-python',
                      'spacenetutilities==3.0', 'keras>=2.2.2',
                      'tensorflow>=1.10.0', 'tensorboard>=1.10.0',
                      'setuptools'],  # TODO: UPDATE
    scripts=[
            'bin/make_np_arrays',
            'bin/train_model',
            'bin/make_predictions0'
        ],

    package_data={
        "": ['*.txt', '*.md', 'Dockerfile']
    },

    python_requires='>=3.6.5',


)
