from setuptools import setup, find_packages

setup(
    name="recsys-project",
    version="0.1.0",
    description="Industrial-grade recommendation system: Two-Tower Recall + DeepFM/DIN Ranking",
    author="recsys-project",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.0",
        "pandas>=2.0",
        "scikit-learn>=1.7",
        "torch>=2.6",
        "tqdm>=4.60",
        "tensorboard>=2.10",
        "pyyaml>=6.0",
        "faiss-cpu>=1.7",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "gradio>=5.0",
    ],
    extras_require={
        "dev": ["jupyter", "ipykernel"],
    },
)
