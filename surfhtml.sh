#!/bin/bash

# Function to display usage information
usage() {
    cat << EOF
Usage: $0 -s SUBJECTS_DIR -h HTMLDIR

This script generates HTML pages to display surface images from FreeSurfer subjects.
It creates hard links to PNG images in the specified HTML directory and generates
HTML files to view them. Images must have been previously created (surfshots.m).

Options:
  -s SUBJECTS_DIR  Path to the FreeSurfer subjects directory containing subject folders
  -h HTMLDIR       Path to the output directory where HTML files and images will be stored

Example:
  $0 -s /path/to/freesurfer/subjects -h /path/to/output/html

Notes:
  - The script skips the 'fsaverage' subject directory.
  - Images are expected in each subject's 'after' subdirectory with names starting with 'bh.'.
  - The HTMLDIR/images subdirectory will be created if it doesn't exist.
  - Existing image links in HTMLDIR/images are not overwritten.

_____________________________________
Anderson M. Winkler
Univ. of Texas Rio Grande Valley
May/2025
http://brainder.org
EOF
    exit 1
}

# Initialize variables
SUBJECTS_DIR=""
HTMLDIR=""

# Parse command-line options
while getopts "s:h:" opt; do
    case $opt in
        s) SUBJECTS_DIR="$OPTARG";;
        h) HTMLDIR="$OPTARG";;
        \?) echo "Invalid option: -$OPTARG" >&2; usage;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage;;
    esac
done

# Check if required arguments are provided
if [ -z "$SUBJECTS_DIR" ] || [ -z "$HTMLDIR" ]; then
    usage
fi

# Make list of subjects
LIST=""
for sub in $(find ${SUBJECTS_DIR} -mindepth 1 -maxdepth 1 -type d|sort) ; do
   if [[ $(basename ${sub}) != fsaverage ]] ; then
      LIST="${LIST} ${sub}"
      asub=${sub}
   fi
done

# Make list of images
ILIST=""
for img in ${asub}/after/bh.*.png ; do
   ILIST="${ILIST} $(basename ${img})"
done

# Hard-link image files to the html folder
mkdir -p ${HTMLDIR}/images
for img in ${ILIST} ; do
   for sub in ${LIST} ; do
      if [[ ! -e ${HTMLDIR}/images/$(basename ${sub}).${img} ]] ; then
         ln ${sub}/after/${img} ${HTMLDIR}/images/$(basename ${sub}).${img}
      fi
   done
done

# Make the HTML pages
for img in ${ILIST} ; do
   HTMLPAGE=${HTMLDIR}/index.${img%.png}.html
   echo "<html><body>" > ${HTMLPAGE}
   for sub in ${LIST} ; do
      echo "<p><img src=images/$(basename ${sub}).${img} width=100%><br>$(basename ${sub}) - ${img}</p>" >> ${HTMLPAGE}
   done
   echo "</body></html>" >> ${HTMLPAGE}
done
