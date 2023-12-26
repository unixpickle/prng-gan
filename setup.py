from setuptools import setup

setup(
    name="prng-gan",
    packages=[
        "prng_gan",
        "prng_gan.scripts",
    ],
    install_requires=[
        "torch",
        "fire",
    ],
    author="Alex ;nichol",
)
