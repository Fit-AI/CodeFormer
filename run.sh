#!/bin/bash

gunicorn -w 2 --threads 2 -b 0.0.0.0:8080 --timeout 6000 service:app 
