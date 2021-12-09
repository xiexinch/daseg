from setuptools import find_packages, setup

version_file = 'daseg/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':

    setup(
        name='daseg',
        version=get_version(),
        description='Domain Adaptation for Semantic Segmentation baseline.',
        # long_description=readme(),
        long_description_content_type='text/markdown',
        author='谢昕辰',
        author_email='xiexinch@outlook.com',
        keywords='domain adaption, semantic segmentation',
        url='http://github.com/xiexinch/daseg_baseline',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        ext_modules=[],
        zip_safe=False)
