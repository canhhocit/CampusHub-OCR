#!/usr/bin/env python3
"""
Generates a legally safe, synthetic specimen CCCD image for end-to-end OCR evaluation.
Contains no real PII; clearly stamped as 'THẺ MẪU / SPECIMEN - CAMPUSHUB TEST ONLY'.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def generate_specimen_front_card(output_path: str = "tests/fixtures/images/specimen_cccd_front.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ID-1 standard resolution: 1012 x 638 px
    width, height = 1012, 638
    
    # Card base: Light cyan / card stock color
    image = Image.new("RGB", (width, height), color=(240, 248, 252))
    draw = ImageDraw.Draw(image)
    
    # Draw card border and decorative header bar
    draw.rectangle([10, 10, width - 10, height - 10], outline=(180, 200, 215), width=3)
    draw.rectangle([15, 15, width - 15, 60], fill=(225, 238, 245))
    
    # Fonts
    font_path = "C:/Windows/Fonts/arial.ttf"
    font_title = ImageFont.truetype(font_path, 22)
    font_sub = ImageFont.truetype(font_path, 16)
    font_bold = ImageFont.truetype(font_path, 20)
    font_card_id = ImageFont.truetype(font_path, 24)
    font_field_label = ImageFont.truetype(font_path, 15)
    font_field_val = ImageFont.truetype(font_path, 17)
    font_watermark = ImageFont.truetype(font_path, 32)
    
    # 1. Header
    draw.text((width // 2, 25), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", fill=(180, 0, 0), font=font_title, anchor="mm")
    draw.text((width // 2, 48), "Độc lập - Tự do - Hạnh phúc", fill=(0, 0, 0), font=font_sub, anchor="mm")
    
    # 2. Card Name
    draw.text((width // 2 + 60, 95), "CĂN CƯỚC CÔNG DÂN", fill=(190, 20, 20), font=font_bold, anchor="mm")
    draw.text((width // 2 + 60, 120), "Citizen Identity Card", fill=(60, 60, 60), font=font_sub, anchor="mm")
    
    # 3. Specimen watermark across card
    draw.text((width // 2, height // 2), "THẺ MẪU / SPECIMEN - TEST ONLY", fill=(210, 220, 225), font=font_watermark, anchor="mm")
    
    # 4. Avatar placeholder
    draw.rectangle([60, 160, 240, 390], outline=(160, 160, 160), fill=(230, 230, 230), width=2)
    draw.text((150, 275), "ẢNH THẺ MẪU", fill=(130, 130, 130), font=font_sub, anchor="mm")
    
    # 5. ID Number
    draw.text((420, 165), "Số / No.: 001098012345", fill=(200, 10, 10), font=font_card_id)
    
    # 6. Full Name
    draw.text((280, 215), "Họ và tên / Full name:", fill=(70, 70, 70), font=font_field_label)
    draw.text((450, 213), "NGUYỄN VĂN MẪU", fill=(10, 10, 10), font=font_field_val)
    
    # 7. Date of birth
    draw.text((280, 260), "Ngày sinh / Date of birth:", fill=(70, 70, 70), font=font_field_label)
    draw.text((480, 258), "12/03/1998", fill=(10, 10, 10), font=font_field_val)
    
    # 8. Gender and Nationality
    draw.text((280, 305), "Giới tính / Sex:", fill=(70, 70, 70), font=font_field_label)
    draw.text((390, 303), "Nam", fill=(10, 10, 10), font=font_field_val)
    draw.text((500, 305), "Quốc tịch / Nationality:", fill=(70, 70, 70), font=font_field_label)
    draw.text((680, 303), "Việt Nam", fill=(10, 10, 10), font=font_field_val)
    
    # 9. Place of origin
    draw.text((280, 350), "Quê quán / Place of origin:", fill=(70, 70, 70), font=font_field_label)
    draw.text((280, 375), "Xã Kim Sơn, Huyện Gia Lâm, TP. Hà Nội", fill=(10, 10, 10), font=font_field_val)
    
    # 10. Place of residence
    draw.text((280, 420), "Nơi thường trú / Place of residence:", fill=(70, 70, 70), font=font_field_label)
    draw.text((280, 445), "Số 45 Đường Giải Phóng, Phường Đồng Tâm", fill=(10, 10, 10), font=font_field_val)
    draw.text((280, 470), "Quận Hai Bà Trưng, TP. Hà Nội", fill=(10, 10, 10), font=font_field_val)
    
    # 11. Expiry date
    draw.text((60, 520), "Có giá trị đến / Date of expiry:", fill=(70, 70, 70), font=font_field_label)
    draw.text((300, 518), "12/03/2038", fill=(10, 10, 10), font=font_field_val)
    
    image.save(output_path, "PNG")
    print(f"Specimen image created successfully at: {output_path}")

if __name__ == "__main__":
    generate_specimen_front_card()
