# Wyoming Piper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [Piper](https://github.com/rhasspy/piper/) text to speech system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_piper)

[Source](https://github.com/home-assistant/addons/tree/master/piper)

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/rhasspy/wyoming-piper.git
cd wyoming-piper
script/setup
```

Run a Wyoming server that Home Assistant can connect to:

``` sh
script/run --voice en_US-lessac-medium --uri 'tcp://0.0.0.0:10200' --data-dir /data --download-dir /data 
```

For a demo web server, make sure to install the `http` dependencies first:

``` sh
script/setup --http
```

Then run in a separate terminal:

``` sh
script/run_http --uri 'tcp://localhost:10200'
```

and visit http://localhost:5000 to test.

## Docker Image

``` sh
docker run -it -p 10200:10200 -v /path/to/local/data:/data rhasspy/wyoming-piper \
    --voice en_US-lessac-medium
```

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/piper)
