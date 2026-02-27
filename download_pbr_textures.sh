#!/bin/bash
# Download PBR texture sets from ambientCG (CC0 license)
# 1K resolution sufficient for 512x512 renders
set -e

TEX_DIR="/mnt/data/yurh/Infinigen-Sim/pbr_textures"
mkdir -p "$TEX_DIR"
cd "$TEX_DIR"

download_tex() {
    local ID=$1
    local CATEGORY=$2
    local OUTDIR="$TEX_DIR/$CATEGORY"
    mkdir -p "$OUTDIR"

    if [ -d "$OUTDIR/$ID" ] && [ "$(ls $OUTDIR/$ID/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] $ID already exists"
        return
    fi

    local URL="https://ambientcg.com/get?file=${ID}_1K-JPG.zip"
    local ZIP="/tmp/${ID}_1K-JPG.zip"

    echo "  Downloading $ID -> $CATEGORY/"
    wget -q -O "$ZIP" "$URL" 2>/dev/null || curl -sL -o "$ZIP" "$URL"

    if [ -f "$ZIP" ] && [ -s "$ZIP" ]; then
        mkdir -p "$OUTDIR/$ID"
        unzip -q -o "$ZIP" -d "$OUTDIR/$ID" 2>/dev/null || true
        rm -f "$ZIP"
        echo "  [OK] $ID: $(ls $OUTDIR/$ID/ | wc -l) files"
    else
        echo "  [FAIL] $ID"
        rm -f "$ZIP"
    fi
}

echo "=== Downloading PBR Textures from ambientCG ==="

# Wood textures (most common material)
echo "Wood textures:"
download_tex "Wood051" "wood"      # dark oak
download_tex "Wood060" "wood"      # light pine
download_tex "Wood049" "wood"      # walnut
download_tex "WoodFloor040" "wood" # engineered/plywood

# Metal textures
echo "Metal textures:"
download_tex "Metal032" "metal"    # brushed steel
download_tex "Metal034" "metal"    # galvanized
download_tex "Metal008" "metal"    # aluminum
download_tex "Metal012" "metal"    # worn/patina

# Plastic textures
echo "Plastic textures:"
download_tex "Plastic012" "plastic"  # smooth matte
download_tex "Plastic006" "plastic"  # rough
download_tex "Plastic010" "plastic"  # textured

# Ceramic/Porcelain
echo "Ceramic textures:"
download_tex "Tiles099" "ceramic"    # white ceramic
download_tex "Tiles074" "ceramic"    # porcelain tile

# Fabric/Leather
echo "Fabric textures:"
download_tex "Fabric051" "fabric"    # woven
download_tex "Fabric038" "fabric"    # cotton
download_tex "Leather037" "leather"  # brown leather
download_tex "Leather025" "leather"  # dark leather

# Rubber/Foam
echo "Rubber textures:"
download_tex "Rubber004" "rubber"    # black rubber

# Stone/Concrete/Marble
echo "Stone textures:"
download_tex "Rock034" "stone"       # granite
download_tex "Marble006" "marble"    # white marble
download_tex "Concrete034" "concrete" # concrete

# Paper/Cardboard
echo "Paper textures:"
download_tex "Paper004" "paper"      # white paper

echo ""
echo "=== Download complete ==="
echo "Textures at: $TEX_DIR"
find "$TEX_DIR" -name "*.jpg" | wc -l
echo "total texture files"
