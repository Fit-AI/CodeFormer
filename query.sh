#!/bin/bash

URL=http://localhost:8080/codeformer
URL=http://35.225.179.110:8090/codeformer
JSON="test.json"

curl -X POST $URL                       \
  -H "Content-Type: application/json"   \
  --data "@$JSON" | jq -Rsa .	
