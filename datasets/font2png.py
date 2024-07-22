#!/usr/bin/env python
import argparse, glob, io, os

from PIL import Image, ImageFont, ImageDraw

DEFAULT_LABEL_FILE = 'characters/50characters.txt'
DEFAULT_FONTS_DIR = 'fonts/source'
DEFAULT_OUTPUT_DIR = '2_DCGAN/datasets'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def font2img(labels, fonts_dir, output_dir, start_idx):

    list_labels = []
    with open(labels, 'r', encoding='utf-8') as fr:
        for line in fr:
            list_labels.append(line.strip())

    with io.open(labels, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir)
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = sorted(glob.glob(os.path.join(fonts_dir, '*.ttf')))
    for f in fonts:
        filename = os.path.basename(f)
        filename_without_extension = os.path.splitext(filename)[0]
        print(filename_without_extension)
              
    # Initialize numbers
    total_count = 0
    prev_count = 0
    font_count = start_idx
    char_no = 0
    
    # Total number of font files is 
    print('total number of fonts are ', len(fonts))

    for character in labels:
        char_no += 1
        
        # Print image count roughly every 1000 images.
        if total_count - prev_count > 1000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            total_count += 1

            image = Image.new('RGB', (IMAGE_WIDTH,IMAGE_HEIGHT), (255, 255, 255))
            w, h = image.size
                
            drawing = ImageDraw.Draw(image)
            font = ImageFont.truetype(font, 170)

            box = None
            new_box = drawing.textbbox((0, 0), character, font)
                
            new_w = new_box[2] - new_box[0]
            new_h = new_box[3] - new_box[1]
                
            box = new_box
            w = new_w
            h = new_h
                
            x = (IMAGE_WIDTH - w)//2 - box[0]
            y = (IMAGE_HEIGHT - h)//2 - box[1]

            drawing.text((x,y), character, fill=(0), font=font) 
            file_string = '{}_{}.png'.format(font_count,hex(ord(character))[2:].upper())
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'PNG')
            font_count += 1
            
        font_count = start_idx
    char_no = 0
            
    print('Finished generating {} images.'.format(total_count))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--label_file', type=str, dest='labels', default=DEFAULT_LABEL_FILE, help='File containing newline delimited labels.')
    parser.add_argument('--font_dir', type=str, dest='fonts_dir', default=DEFAULT_FONTS_DIR, help='Directory of ttf fonts to use.')
    parser.add_argument('--output_dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR, help='Output directory to store generated images.')
    parser.add_argument('--start_idx', type=int, dest='start_idx', default=0, help='Font count to include in the file name.')

    args = parser.parse_args()

    font2img(args.labels, args.fonts_dir, args.output_dir, args.start_idx)
