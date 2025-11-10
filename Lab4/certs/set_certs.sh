#!/bin/bash

# Set your policy name (change this)
defaultPolicyName="Lab4_Carpolicy"

# List of thing names to process
for i in {1442..1661}
do
  thingName="IOTCar$i"
  echo "\nProcessing $thingName..."

  # 1. Create certificate and capture output
  output=$(aws iot create-keys-and-certificate --set-as-active)
  folder="IOTCar$i"
  mkdir -p "$folder"
  certificateArn=$(echo "$output" | jq -r '.certificateArn')
  certificatePem=$(echo "$output" | jq -r '.certificatePem')
  privateKey=$(echo "$output" | jq -r '.keyPair.PrivateKey')
  publicKey=$(echo "$output" | jq -r '.keyPair.PublicKey')

  # 2. Save keys and certs to separate files (name them per car/device)
  echo "$certificatePem" > "$folder/Car${i}-cert.pem"
  echo "$privateKey" > "$folder/Car${i}-priv.key"
  echo "$publicKey"  > "$folder/Car${i}-pub.key"

  # 3. Attach policy to certificate
  aws iot attach-policy --policy-name "$defaultPolicyName" --target "$certificateArn"

  # 4. Attach certificate to the thing
  aws iot attach-thing-principal --thing-name "$thingName" --principal "$certificateArn"

done

