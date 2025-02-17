# Use the latest Red Hat UBI image
FROM registry.access.redhat.com/ubi8/python-311

# disable python stdout buffer
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY pyproject.toml README.md /app/

# Set up permissions
USER root
RUN chown -R 1001:0 /app
USER 1001

# install project
RUN pip install --no-cache-dir .

# Copy the application code
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
