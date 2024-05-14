#!/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Please specify 'ssh' or 'https' as the argument."
    exit 1
fi

# Function to convert HTTPS to SSH URL
convert_to_ssh() {
    echo "$1" | sed -E 's#https://([^/]+)/(.+)#git@\1:\2#g'
}

# Function to convert SSH to HTTPS URL
convert_to_https() {
    echo "$1" | sed -E 's#git@([^:]+):(.+)#https://\1/\2#g'
}

# Iterate over each submodule
git config --file .gitmodules --get-regexp '^submodule\..*\.path$' | while read path_key submodule_path; do
    # Extract the last part of the submodule path
    submodule_name=$(basename "$submodule_path")

    # Extract the current URL of the submodule
    url_key="submodule.${submodule_name}.url"
    current_url=$(git config --file .gitmodules --get "$url_key")

    case "$1" in
        ssh)
            new_url=$(convert_to_ssh "$current_url")
            ;;
        https)
            new_url=$(convert_to_https "$current_url")
            ;;
        *)
            echo "Invalid argument. Please specify 'ssh' or 'https'."
            exit 1
            ;;
    esac

    # Update the submodule URL
    if [ "$current_url" != "$new_url" ]; then
        echo "Changing submodule URL from $current_url to $new_url"
        git config --file .gitmodules "submodule.${submodule_name}.url" "$new_url"
    fi
done

git submodule sync
