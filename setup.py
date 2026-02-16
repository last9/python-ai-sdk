"""
Setup script for last9-genai

This setup.py is provided for backwards compatibility.
Modern installations should use pyproject.toml.
"""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="last9-genai",
    version="1.0.0",
    author="Last9 Inc.",
    author_email="hello@last9.io",
    description="Last9 observability attributes for OpenTelemetry GenAI spans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/last9/python-ai-sdk",
    project_urls={
        "Bug Reports": "https://github.com/last9/python-ai-sdk/issues",
        "Source": "https://github.com/last9/python-ai-sdk",
        "Documentation": "https://github.com/last9/python-ai-sdk#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
    ],
    extras_require={
        "otlp": [
            "opentelemetry-exporter-otlp-proto-grpc>=1.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
        ],
        "examples": [
            "anthropic>=0.3.0",
            "openai>=1.0.0",
            "langchain>=0.1.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    keywords="opentelemetry genai llm observability last9 anthropic openai langchain ai monitoring cost-tracking",
)
