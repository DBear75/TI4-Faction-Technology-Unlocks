import argparse
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import textwrap
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import time

_EPS = 1e-12

def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Faster vectorized RGB [0,1] -> HSL [0,1]
    rgb shape: (..., 3)
    Returns: h, s, l
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    l = 0.5 * (maxc + minc)

    # Saturation
    denom = 1.0 - np.abs(2.0 * l - 1.0)
    s = np.divide(delta, denom, out=np.zeros_like(delta), where=denom > _EPS)

    # Hue
    h = np.zeros_like(delta)

    nz = delta > _EPS
    inv_delta = np.divide(1.0, delta, out=np.zeros_like(delta), where=nz)

    # Use >= to match max-channel tie behavior consistently
    rmask = nz & (r >= g) & (r >= b)
    gmask = nz & (g > r) & (g >= b)
    bmask = nz & (b > r) & (b > g)

    h[rmask] = ((g[rmask] - b[rmask]) * inv_delta[rmask]) % 6.0
    h[gmask] = ((b[gmask] - r[gmask]) * inv_delta[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) * inv_delta[bmask]) + 4.0
    h /= 6.0

    return h, s, l


def _hue_to_rgb(m1: np.ndarray, m2: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Faster branch-reduced helper for HSL->RGB.
    """
    h = np.mod(h, 1.0)

    a = m1 + (m2 - m1) * 6.0 * h
    b = m2
    c = m1 + (m2 - m1) * 6.0 * (2.0 / 3.0 - h)
    d = m1

    return np.select(
        [
            h < (1.0 / 6.0),
            h < 0.5,
            h < (2.0 / 3.0),
        ],
        [a, b, c],
        default=d,
    )


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Faster vectorized HSL [0,1] -> RGB [0,1]
    h, s, l shape: (...,)
    Returns rgb shape: (..., 3)
    """
    m2 = np.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
    m1 = 2.0 * l - m2

    r = _hue_to_rgb(m1, m2, h + 1.0 / 3.0)
    g = _hue_to_rgb(m1, m2, h)
    b = _hue_to_rgb(m1, m2, h - 1.0 / 3.0)

    rgb = np.stack((r, g, b), axis=-1)

    # Achromatic override
    achromatic = s <= _EPS
    if np.any(achromatic):
        rgb[achromatic] = l[achromatic][..., None]

    return np.clip(rgb, 0.0, 1.0)


def composite_source_over(
    base_rgb: np.ndarray,
    base_a: np.ndarray,
    top_rgb: np.ndarray,
    top_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard source-over alpha compositing for straight-alpha images.
    Optimized to reduce temporary arrays.
    """
    one_minus_top_a = 1.0 - top_a
    out_a = top_a + base_a * one_minus_top_a

    out_premul = (
        top_rgb * top_a[..., None] +
        base_rgb * (base_a * one_minus_top_a)[..., None]
    )

    out_rgb = np.divide(
        out_premul,
        out_a[..., None],
        out=np.zeros_like(out_premul),
        where=out_a[..., None] > _EPS
    )

    return np.clip(out_rgb, 0.0, 1.0), np.clip(out_a, 0.0, 1.0)


def apply_hsl_color_overlay(
    base_img: Image.Image,
    color_img: Image.Image,
    opacity: float = 1.0,
) -> Image.Image:
    """
    Apply GIMP-like HSL Color blending:
      - hue from color image
      - saturation from color image
      - lightness from base image
    Then alpha-composite over the base using the color image alpha * opacity.
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))

    base_rgba = np.asarray(base_img.convert("RGBA"), dtype=np.float32) / 255.0
    color_rgba = np.asarray(color_img.convert("RGBA"), dtype=np.float32) / 255.0

    base_rgb = base_rgba[..., :3]
    base_a = base_rgba[..., 3]
    color_rgb = color_rgba[..., :3]
    color_a = color_rgba[..., 3] * opacity

    # Early exit: fully transparent overlay
    if not np.any(color_a > _EPS):
        return base_img.convert("RGBA")

    _, _, base_l = rgb_to_hsl(base_rgb)
    color_h, color_s, _ = rgb_to_hsl(color_rgb)

    blended_rgb = hsl_to_rgb(color_h, color_s, base_l)
    out_rgb, out_a = composite_source_over(base_rgb, base_a, blended_rgb, color_a)

    out = np.empty_like(base_rgba)
    out[..., :3] = out_rgb
    out[..., 3] = out_a

    return Image.fromarray(np.round(out * 255.0).astype(np.uint8), mode="RGBA")

highlight_terms = {
    "SUSTAIN",
    "DAMAGE",
    "SPACE",
    "CANNON",
    "ANTI-FIGHTER",
    "BARRAGE",
    "BOMBARDMENT"
}

def wrap_text_by_pixel(text, font, max_width, draw, highlight_terms=None):
    if highlight_terms is None:
        highlight_terms = set()

    # Tokenize words
    raw_words = text.split()
    i = 0
    tokens = []

    while i < len(raw_words):
        matched = False
        for span in [3, 2, 1]:
            if i + span <= len(raw_words):
                phrase_words = raw_words[i:i+span]
                phrase = " ".join(phrase_words)
                if phrase in highlight_terms:
                    tokens.append(phrase)
                    i += span
                    matched = True
                    break
        if not matched:
            tokens.append(raw_words[i])
            i += 1

    # Now wrap using token list
    lines = []
    current_line = ""
    for token in tokens:
        test_line = current_line + (" " if current_line else "") + token
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = token

    if current_line:
        lines.append(current_line)

    return lines


def build_color_map(mask_img: Image.Image, faction_color: tuple[int, int, int]) -> Image.Image:
    arr = np.asarray(mask_img.convert("RGBA"), dtype=np.uint8)
    out = np.zeros_like(arr)

    white_mask = (
        (arr[..., 0] == 255) &
        (arr[..., 1] == 255) &
        (arr[..., 2] == 255)
    )

    out[white_mask, 0] = faction_color[0]
    out[white_mask, 1] = faction_color[1]
    out[white_mask, 2] = faction_color[2]
    out[white_mask, 3] = 255

    return Image.fromarray(out, mode="RGBA")


def draw_mixed_font_line(
    draw, line, y, x_start, normal_font, special_font, font_unlock, font_size, font_color='white', center=False, highlight_terms=None
):
    if highlight_terms is None:
        highlight_terms = {
            "SUSTAIN",
            "DAMAGE",
            "SPACE",
            "CANNON",
            "ANTI-FIGHTER",
            "BARRAGE",
            "BOMBARDMENT"
        }

    words = line.split()
    if center:
        total_width = 0
        for word in words:
            if word in highlight_terms:
                total_width += draw.textlength(word + " ", font=special_font)
            else:
                total_width += draw.textlength(word + " ", font=normal_font)
        x_cursor = (draw.im.size[0] - total_width) // 2
    else:
        x_cursor = x_start
    i = 0
    dy_special = -font_size * 0.07
    while i < len(words):
        matched = False
        for span in [3, 2, 1]:
            if i + span <= len(words):
                phrase_words = words[i:i+span]
                phrase = " ".join(phrase_words)
                if phrase in highlight_terms:
                    draw.text((x_cursor, y + dy_special), phrase, font=special_font, fill=font_color)
                    x_cursor += draw.textlength(phrase + " ", font=special_font)
                    i += span
                    matched = True
                    break
                elif phrase == "UNLOCK:":
                    draw.text((x_cursor, y), phrase, font=font_unlock, fill=font_color)
                    x_cursor += draw.textlength(phrase + " ", font=font_unlock)
                    i += span
                    matched = True
                    break
        if not matched:
            word = words[i]
            draw.text((x_cursor, y), word, font=normal_font, fill=font_color)
            x_cursor += draw.textlength(word + " ", font=normal_font)
            i += 1
    
    return

from PIL import Image, ImageDraw
import numpy as np


def draw_gradient_text(
    base_image,
    position,
    text,
    font,
    start_color,
    end_color=(255, 255, 255),
    anchor="mm",
):
    """
    Draw horizontal gradient text onto base_image.
    """
    x, y = position
    # Measure exact glyph bounds
    temp_draw = ImageDraw.Draw(base_image)
    bbox = temp_draw.textbbox((0, 0), text, font=font, anchor=None)
    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top

    if w <= 0 or h <= 0:
        return

    # Build text mask with proper glyph offset compensation
    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text((-left, -top), text, font=font, fill=255)

    # Build horizontal RGB gradient
    t = np.linspace(0.0, 1.0, w, dtype=np.float32) ** 0.6
    grad_row = (
        np.array(start_color, dtype=np.float32)[None, :]
        + (np.array(end_color, dtype=np.float32) - np.array(start_color, dtype=np.float32))[None, :] * t[:, None]
    )
    grad = np.broadcast_to(grad_row[None, :, :], (h, w, 3)).astype(np.uint8)
    grad_img = Image.fromarray(grad, "RGB")

    # Anchor handling
    if anchor == "mm":
        paste_x = int(round(x - w / 2))
        paste_y = int(round(y - h / 2))
    elif anchor == "lm":
        paste_x = int(round(x))
        paste_y = int(round(y - h / 2))
    elif anchor == "rm":
        paste_x = int(round(x - w))
        paste_y = int(round(y - h / 2))
    elif anchor == "lt":
        paste_x = int(round(x))
        paste_y = int(round(y))
    else:
        raise ValueError(f"Unsupported anchor: {anchor}")

    base_image.paste(grad_img, (paste_x, paste_y), mask)

def make_cards(
    data,
    front_background,
    back_background,
    color_maps,
    faction_symbols,
    font_header,
    font_body,
    font_unlock,
    font_special,
    highlight_terms,
    header_font_size=100,
    body_font_size=70,
    gamecrafter=False
):
    # All constant values
    if gamecrafter:
        title_y = 100
        extra_y_inc = 5
        border = 225
        edge_distance = 125
        faction_symbol_size = 125
        generated_images_loc = "generatedImages/gamecrafter"
    else:
        title_y = 225
        extra_y_inc = 10
        border = 450
        edge_distance = 250
        faction_symbol_size = 250
        generated_images_loc = "generatedImages"

    for index, row in data.iterrows():
        faction = row['Faction']
        expansion = row['Expansion']
        title = row['Title']
        unlock_condition = "UNLOCK: " + row['Unlock Condition']
        reward_wording = row['Reward Wording']
        faction_color = row['Faction Color']
        faction_color = tuple(int(faction_color[i:i+2], 16) for i in (0, 2, 4))

        # Create a new image for the card
        card_image = Image.new('RGBA', front_background.size)
        card_image.paste(front_background, (0, 0))

        # Create the back of the card
        back_image = Image.new('RGBA', back_background.size)
        back_image.paste(back_background, (0, 0))

        # Write the title text on the cards
        draw = ImageDraw.Draw(card_image)
        draw_back = ImageDraw.Draw(back_image)

        title_lines = wrap_text_by_pixel(title.upper(), font_header, card_image.width - edge_distance, draw)
        y = title_y
        if len(title_lines) == 1:
            y += header_font_size//2

        center_x = back_image.width // 2
        for line in title_lines:
            draw_gradient_text(
                card_image,
                (center_x, y),
                line,
                font_header,
                faction_color,
                (255, 255, 255),
                anchor="mm"
            )
            draw_gradient_text(
                back_image,
                (center_x, y),
                line,
                font_header,
                faction_color,
                (255, 255, 255),
                anchor="mm"
            )
            y += header_font_size + extra_y_inc

        # Write the unlock condition text on the card centered in the middle of the card
        unlock_lines = wrap_text_by_pixel(unlock_condition, font_body, card_image.width - border, draw, highlight_terms=highlight_terms)
        total_text_height = sum(body_font_size + extra_y_inc for line in unlock_lines) - extra_y_inc
        y = (card_image.height - total_text_height) // 2
        for line in unlock_lines:
            draw_mixed_font_line(draw, line, y, 100, font_body, font_special, font_unlock, body_font_size, center=True, highlight_terms=highlight_terms)
            y += body_font_size + extra_y_inc

        # Write the reward wording text on the back in the middle of the card
        reward_lines = wrap_text_by_pixel(reward_wording, font_body, back_image.width - border, draw_back, highlight_terms=highlight_terms)
        total_text_height = sum(body_font_size + extra_y_inc for line in reward_lines) - extra_y_inc
        y = (back_image.height - total_text_height) // 2
        for line in reward_lines:
            draw_mixed_font_line(draw_back, line, y, 100, font_body, font_special, font_unlock, body_font_size, center=True, highlight_terms=highlight_terms)
            y += body_font_size + extra_y_inc

        # Apply color maps as an HSL Color overlay to back
        back_image = apply_hsl_color_overlay(back_image, color_maps[faction]['card'])
        back_image = apply_hsl_color_overlay(back_image, color_maps[faction]['starfield'])

        # Paste the faction symbol in the bottom left corner of the back
        if faction.lower() in faction_symbols:
            symbol = faction_symbols[faction.lower()]
            symbol = symbol.resize((faction_symbol_size, faction_symbol_size), Image.LANCZOS)
            x_loc = card_image.width - symbol.width//2 - edge_distance
            y_loc = card_image.height - symbol.height//2 - edge_distance
            card_image.paste(
                symbol, 
                (x_loc, y_loc),
                symbol
            )
            x_loc = - symbol.width//2 + edge_distance
            back_image.paste(
                symbol, 
                (x_loc, y_loc),
                symbol
            )
            
        else:
            print(f"Warning: No faction symbol found for {faction}")

        # Save the card images
        card_image.save(f"{generated_images_loc}/{expansion}/fronts/{faction.lower()}_{title}-front.png")
        back_image.save(f"{generated_images_loc}/{expansion}/backs/{faction.lower()}_{title}-back.png")


def main():
    # Start Timer
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Generates faction tech unlock cards.')

    parser.add_argument(
        "--input-file",
        type=str,
        help="The csv file that will be used to generate the objectives.",
        default=None
    )

    parser.add_argument(
        "--gamecrafter",
        action='store_true',
        default=False,
        help="Whether to format the cards for GameCrafter printing (600x825) instead of TTS (750x1050)."
    )

    parser.add_argument(
        "--deck-only",
        action='store_true',
        default=False,
        help="Only generate the combined deck images for TTS, skipping individual card images. Implies --tts-mode."
    )

    parser.add_argument("--tts-mode", action='store_true', default=False)

    args = parser.parse_args()

    print("GameCrafter mode:", args.gamecrafter)

    if args.deck_only:
        args.tts_mode = True

    # Read the input CSV file
    data = pd.read_csv(args.input_file)

    # Find list of unique expansions in the data
    expansions = data['Expansion'].unique()

    # Check if the generatedImages folder exists
    if args.gamecrafter:
        generated_images_loc = "generatedImages/gamecrafter"
    else:
        generated_images_loc = "generatedImages/full-resolution"
        
    if not os.path.exists(f"{generated_images_loc}/gamecrafter"):
        os.makedirs(f"{generated_images_loc}/gamecrafter")

    for expansion in expansions:
        # Check if the fronts folder exists
        if not os.path.exists(f"{generated_images_loc}/{expansion}/fronts"):
            # Create the fronts folder
            os.makedirs(f"{generated_images_loc}/{expansion}/fronts")

        if not os.path.exists(f"{generated_images_loc}/{expansion}/backs"):
            # Create the backs folder
            os.makedirs(f"{generated_images_loc}/{expansion}/backs")
    if args.gamecrafter:
        front_background = Image.open("backgrounds/front-gamecrafter.png")
        back_background = Image.open("backgrounds/back-gamecrafter.png")
        card_color_area = Image.open("color-areas/card-outline-gamecrafter.png")
        starfield_color_area = Image.open("color-areas/starfield-gamecrafter.png")
        header_font_size = 50 
        body_font_size = 35
        color_maps_loc = f"{generated_images_loc}/color-maps-gamecrafter"
    else:
        front_background = Image.open("backgrounds/front.png")
        back_background = Image.open("backgrounds/back.png")
        card_color_area = Image.open("color-areas/card-outline.png")
        starfield_color_area = Image.open("color-areas/starfield.png")
        header_font_size = 100
        body_font_size = 70
        color_maps_loc = f"{generated_images_loc}/color-maps"
    
    # Define the font and size for the text
    font_header = ImageFont.truetype("SliderTI-_.otf", header_font_size)
    font_body = ImageFont.truetype("MyriadPro-Regular.otf", body_font_size)
    font_unlock = ImageFont.truetype("MYRIADPRO-BOLDIT.otf", body_font_size)
    font_special = ImageFont.truetype("SliderTI-_.otf", int(0.9*body_font_size))

    # pre-load faction symbols by looking in faction-icons folder and mapping file names to faction names
    faction_symbols = {}
    for filename in os.listdir("faction-icons"):
        if filename.endswith(".png"):
            faction_name = os.path.splitext(filename)[0]
            faction_symbols[faction_name] = Image.open(os.path.join("faction-icons", filename))

    # pre-build color maps by filling the white areas of the color area images with the faction color, and leaving the rest transparent
    color_maps = {}
    for faction in data['Faction'].unique():
        if os.path.exists(f"{color_maps_loc}/{faction}_card.png") and os.path.exists(f"{color_maps_loc}/{faction}_starfield.png"):
            card_color_map = Image.open(f"{color_maps_loc}/{faction}_card.png")
            starfield_color_map = Image.open(f"{color_maps_loc}/{faction}_starfield.png")
            color_maps[faction] = {
                'card': card_color_map,
                'starfield': starfield_color_map
            }
            continue
        # Color is in for FFFFFF format, we need to convert it to an (R, G, B) tuple
        faction_color = data.loc[data['Faction'] == faction, 'Faction Color'].iloc[0]
        faction_color = tuple(int(faction_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create color map for card
        card_color_map = build_color_map(card_color_area, faction_color)
        starfield_color_map = build_color_map(starfield_color_area, faction_color)

        # Save the color maps for future use
        if not os.path.exists(color_maps_loc):
            os.makedirs(color_maps_loc)
        
        card_color_map.save(f"{color_maps_loc}/{faction}_card.png")
        starfield_color_map.save(f"{color_maps_loc}/{faction}_starfield.png")
        color_maps[faction] = {
            'card': card_color_map,
            'starfield': starfield_color_map
        }

    if not args.deck_only:
        max_workers = min(max(1, cpu_count() - 2), data.shape[0]//16)
        print(f"Using {max_workers} workers for parallel card generation...")
        # Split data into chunks for each worker
        data_chunks = np.array_split(data, max_workers)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk in data_chunks:
                futures.append(executor.submit(
                    make_cards,
                    chunk,
                    front_background,
                    back_background,
                    color_maps,
                    faction_symbols,
                    font_header,
                    font_body,
                    font_unlock,
                    font_special,
                    highlight_terms,
                    header_font_size,
                    body_font_size,
                    args.gamecrafter
                ))

            # Wait for all workers to finish
            for future in futures:
                future.result()
        
        # End Timer
        end_time = time.time()
        print(f"Card generation completed in {end_time - start_time:.2f} seconds.")

    # Generate Deck Images for TTS
    if args.tts_mode:
        # Check if deck folder exists
        if not os.path.exists(f"{generated_images_loc}/decks"):
            os.makedirs(f"{generated_images_loc}/decks")

        x_deck = 10
        for expansion in expansions:
            fronts = []
            backs = []
            for index, row in data[data['Expansion'] == expansion].iterrows():
                faction = row['Faction']
                title = row['Title']
                front_path = f"{generated_images_loc}/{expansion}/fronts/{faction.lower()}_{title}-front.png"
                back_path = f"{generated_images_loc}/{expansion}/backs/{faction.lower()}_{title}-back.png"
                fronts.append(Image.open(front_path))
                backs.append(Image.open(back_path))
            
            deck_size = (fronts[0].width * x_deck, fronts[0].height * ((len(fronts) + x_deck - 1) // x_deck))
            # Combine all fronts into a single image
            combined_front = Image.new('RGBA', deck_size)
            combined_back = Image.new('RGBA', deck_size)
            for i, (front, back) in enumerate(zip(fronts, backs)):
                x = (i % x_deck)*front.width
                y = (i // x_deck)*front.height
                combined_front.paste(front, (x, y))
                combined_back.paste(back, (x, y))
            
            if not args.gamecrafter:
                # Halve the deck size and remove alpha channel
                combined_front = combined_front.resize((combined_front.width // 2, combined_front.height // 2), Image.LANCZOS).convert("RGB")
                combined_back = combined_back.resize((combined_back.width // 2, combined_back.height // 2), Image.LANCZOS).convert("RGB")
            else:
                # Just remove alpha channel for GameCrafter
                combined_front = combined_front.convert("RGB")
                combined_back = combined_back.convert("RGB")

            combined_front.save(f"{generated_images_loc}/decks/{expansion}_deck_front.jpg", quality=50)
            combined_back.save(f"{generated_images_loc}/decks/{expansion}_deck_back.jpg", quality=50)
    

if __name__ == "__main__":
    main()




    