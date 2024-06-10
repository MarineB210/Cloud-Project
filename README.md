The first step is to build the docker image. To do this, you need to run the following command in the terminal:

```bash
docker build -t multimedia-app .
```

After building the image, you can run the container with the following command:

```bash
docker run -p 5000:5000 multimedia-app
```

The application will be available at [http://localhost:5000.](http://163.172.234.165:5000/)


You need to register to the application to use it. You can create your own credentials. After registering, you can login to the application and use it.