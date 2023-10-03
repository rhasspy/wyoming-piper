# Wyoming Piper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [Piper](https://github.com/rhasspy/piper/) text to speech system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_piper)

[Source](https://github.com/home-assistant/addons/tree/master/piper)

## Docker Image

``` sh
docker run -it -p 10200:10200 -v /path/to/local/data:/data rhasspy/wyoming-piper \
    --voice en_US-lessac-medium
```

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/piper)
