# Use an official Python runtime as a parent image with specific version
FROM python:3.7.3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY .. /app


# Upgrade pip version
RUN pip install --upgrade pip

# Install pip dependencies from setup.py
RUN pip install -r requirements.txt


# Run app.py when the container launches
CMD ["python", "run_training.py"]