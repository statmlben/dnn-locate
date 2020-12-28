from setuptools import setup, find_packages

with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

if __name__ == "__main__":
    setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='dnn-locate',
        url='https://github.com/statmlben/dnn-locate',
        author=['Ben Dai', 'Chunlin Li'],
        author_email='bdai@umn.edu, li000007@umn.edu',
        # Needed to actually package something
        packages=['dnn-locate'],
        # Needed for dependencies
        install_requires=['numpy', 'keras', 'pandas', 'tensorflow>=2.3.1', 'scipy', 'sklearn', 'matplotlib'],
        # *strongly* suggested for sharing
        version='0.1',
        # The license can be anything you like
        license='MIT',
        description='dnn-locate is a Python module for discriminative features localization based on neural networks.',
        # cmdclass={"build_ext": build_ext},
        # We will also need a readme eventually (there will be a warning)
        long_description_content_type='text/x-rst',
        long_description=LONG_DESCRIPTION,
    )
