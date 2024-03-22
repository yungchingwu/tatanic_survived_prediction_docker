# ML Server and Web Server Dockerized Application

This repository contains a Dockerized application that consists of a machine learning (ML) server and a web server. The ML server handles data processing and machine learning tasks, while the web server serves a user interface for interacting with the ML server.

## Directory Structure

- `ml_server`: Contains files related to the ML server.
  - `ml_server_data`: Additional resources specific to the ML server.
- `web_server`: Contains files related to the web server.
  - `web_server_data`: Resources for the web server, such as static files and templates.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `docker-compose.yaml`: Configuration file for Docker Compose.
- `mysql`: Directory for MySQL database initialization script.
- `README.md`: You are currently reading this file.

## Prerequisites

Ensure you have Docker and Docker Compose installed on your system to run the application.

## Installation and Setup

1. Clone this repository to your local machine.
2. Navigate to the root directory of the repository.
3. Run the following command to build and start the Docker containers:

```bash
docker-compose up --build
```

This command will build the Docker images and start the containers for both the ML server and the web server.

## Usage

Once the Docker containers are running, you can access the web server at `http://localhost:80` in your web browser. From the web interface, you can interact with the ML server to perform various machine learning tasks.

## Customization

- **ML Server**: Customize the machine learning tasks, data processing, and models in the `ml_server` directory.
- **Web Server**: Customize the user interface, static files, and templates in the `web_server` directory.
- **Database**: If you need to modify the database schema or initialization script, make changes in the `mysql` directory.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
