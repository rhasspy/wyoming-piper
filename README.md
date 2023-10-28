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

Install Piper
```sh
curl -L -s "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz" | tar -zxvf - -C /usr/share
```

Run a server that anyone can connect to:

``` sh
script/run --piper '/usr/share/piper/piper' --voice en_US-lessac-medium --uri 'tcp://0.0.0.0:10200' --data-dir /data --download-dir /data 
```

## Docker Image

``` sh
docker run -it -p 10200:10200 -v /path/to/local/data:/data rhasspy/wyoming-piper \
    --voice en_US-lessac-medium
```

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/piper)
