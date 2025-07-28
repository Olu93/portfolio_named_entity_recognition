@echo off

REM Build and run the NER API Docker container

echo Building Docker image...
docker build -t ner-api .

echo Running Docker container...
docker run -d ^
  --name ner-api-container ^
  -p 8000:8000 ^
  -v "%cd%/files:/app/files" ^
  --restart unless-stopped ^
  ner-api

echo Container started! API available at http://localhost:8000
echo API documentation available at http://localhost:8000/docs
echo.
echo To stop the container: docker stop ner-api-container
echo To remove the container: docker rm ner-api-container
echo To view logs: docker logs ner-api-container

pause 