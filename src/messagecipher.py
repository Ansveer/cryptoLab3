import numpy as np
import os
import json
import argparse
import struct
import sys
from PIL import Image
from bitarray import bitarray


def bytes_to_bits(bytes):
    bits = []
    for byte in bytes:
        for i in range(8):
            bits.append(int(format(byte, '08b')[i]))
    return bits


def bits_to_bytes(bits):
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = ""
        for j in range(8):
            byte += str(bits[i + j])
        bytes_list.append(int(byte, 2))
    return bytes(bytes_list)


def keystreamGenLCG(n, key):
    # X_0 - начальное значение (0 <= X_0 < m)
    # m >= 2
    # a - множитель, c - приращение (0 <= a < m) (0 <= c < m)
    X_0 = key
    m = 2**64
    a = 43252341515252341
    c = 13243223452435

    keystream = []
    keystream.append(X_0)
    for i in range(1, n):
        X_n = (a*keystream[i - 1] + c) % m
        keystream.append(X_n)

    return keystream


def keystreamGen(key):
    # X_0 - начальное значение (0 <= X_0 < m)
    # m >= 2
    # a - множитель, c - приращение (0 <= a < m) (0 <= c < m)
    X_n1 = key
    m = 2**64
    a = 43252341515252341
    c = 13243223452435

    X_n = (a * X_n1 + c) % m

    return X_n


def correctKeystream(n, keystream, m):
    tmp = []
    for i in range(n):
        tmp.append(keystream[i] % m)

    return tmp


def xorStream(plaintext, keystream, length):
    arr = []
    for i in range(length):
        arr.append(plaintext[i] ^ keystream[i])

    return arr


def change_bit(pixel_value, bit):
    return pixel_value & 0xFE | bit


def LSB(encode_message, img, height, width, channels):
    pixels = img
    i = 0
    endEmbed = False
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                if i >= len(encode_message):
                    endEmbed = True
                    break
                old_pixel = pixels[x][y][channel]
                new_pixel = change_bit(old_pixel, encode_message[i])
                pixels[x][y][channel] = new_pixel
                i += 1
            if endEmbed:
                break
        if endEmbed:
            break
    return pixels


def getBits(length, img, height, width, channels):
    pixels = img
    extractedBits = []
    endFor = False
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                if (x + y + channel) >= length:
                    endFor = True
                    break

                bit = pixels[x][y][channel] & 1
                extractedBits.append(bit)
            if endFor:
                break
        if endFor:
            break
    return extractedBits


if __name__ == "__main__":
    # [0:8] - IV; [8: 12] - length; [12: len(message)] - message
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["encrypt", "decrypt"], required=True)
    parser.add_argument('--message')
    parser.add_argument('--password', required=True)
    parser.add_argument('--in_img')
    parser.add_argument('--out_img')
    parser.add_argument('--LSB', choices=["random", "classic"])
    args = parser.parse_args()

    if args.mode == 'encrypt':
        if args.message and args.password:
            # Берём сообщение из файла
            with open(f"{args.message}", 'r', encoding='utf-8') as file:
                message = file.read()

            message = ""
            for i in range(16378):
                message += "a\n"
            with open('../results/test_message.txt', 'w', encoding='utf-8') as file:
                file.write(message)
            # IV длиной 8 байт
            iv_bytes = os.urandom(8)
            # Длина (число) в 4 байтах
            length_bytes = struct.pack('>I', len(message))
            # Переводим сообщение в байты
            message_bytes = message.encode("utf-8")

            # На основе пароля и IV получается симметричный ключ (LCG)
            key = int(args.password.encode('utf-8').hex() + iv_bytes.hex(), 16)
            keystream = keystreamGenLCG(len(message), key)
            keystream = correctKeystream(len(message), keystream, 255)

            # xor операция между сообщением и keystream
            ciphertext = xorStream(message_bytes, keystream, len(message))
            # Итоговое зашифрованное сообщение из IV, длины сообщения и самого сообщения
            encrypted_message = iv_bytes + length_bytes + bytes(ciphertext)

            meta = {
                "IV": f"{iv_bytes.hex()}",
            }

            with open('../results/encrypted_message.txt', 'w', encoding='utf-8') as file:
                file.write(encrypted_message.hex())

            with open("../results/META.json", "w", encoding="utf-8") as file:
                json.dump(meta, file, ensure_ascii=False)

        if args.in_img and args.out_img and args.password and args.LSB:
            img = Image.open(f"../imgs/{args.in_img}")
            pixels = np.array(img)
            pixels_flat = pixels.flatten()
            width, height, channels = pixels.shape

            with open('../results/encrypted_message.txt', 'r', encoding='utf-8') as file:
                message = bytes.fromhex(file.read())

            need_bits = 8 * len(message)

            if need_bits > width * height * 3:
                print("Сообщение слишком большое")
                sys.exit(0)

            encode_message = bytes_to_bits(message)

            if args.LSB == "random":
                used_positions = [0] * width * height * 3
                key = int(args.password.encode('utf-8').hex(), 16)
                keystream = []
                m = width * height * 3
                tmp = keystreamGen(key)
                keystream.append(tmp)
                used_positions[tmp % m] = 1
                for i in range(len(encode_message) - 1):
                    tmp = keystreamGen(tmp)
                    while used_positions[tmp % m]:
                        tmp = keystreamGen(tmp)
                    keystream.append(tmp)
                    used_positions[tmp % m] = 1
                keystream = correctKeystream(len(encode_message), keystream, width * height * 3)

                for i in range(len(keystream)):
                    old_pixel = pixels_flat[keystream[i]]
                    new_pixel = change_bit(old_pixel, encode_message[i])
                    pixels_flat[keystream[i]] = new_pixel

                result_img = Image.fromarray(pixels_flat.reshape((width, height, channels)))
                result_img.save(f"../imgs/{args.out_img}")
            elif args.LSB == "classic":
                img = LSB(encode_message, pixels, height, width, channels)
                result_img = Image.fromarray(img)
                result_img.save(f"../imgs/{args.out_img}")
            else:
                print("Выбран некорректный LSB режим")
                sys.exit(0)

    elif args.mode == 'decrypt':
        if args.in_img and args.password and args.LSB:
            img = Image.open(f"../imgs/{args.in_img}")
            pixels = np.array(img)
            pixels_flat = pixels.flatten()
            width, height, channels = pixels.shape

            with open("../results/META.json", "r", encoding="utf-8") as file:
                meta = json.load(file)

            iv_bytes = bytes.fromhex(meta['IV'])

            if args.LSB == "random":
                key_img = int(args.password.encode('utf-8').hex(), 16)

                used_positions = [0] * width * height * 3
                keystream_img = []
                m = width * height * 3
                tmp = keystreamGen(key_img)
                keystream_img.append(tmp)
                used_positions[tmp % m] = 1
                for i in range(12*8 - 1):
                    tmp = keystreamGen(tmp)
                    while used_positions[tmp % m]:
                        tmp = keystreamGen(tmp)
                    keystream_img.append(tmp)
                    used_positions[tmp % m] = 1
                last_keystream = keystream_img[-1]
                keystream_img = correctKeystream(12*8, keystream_img, width * height * 3)

                tmp = []
                for i in keystream_img:
                    tmp.append(pixels_flat[i] & 1)

                extracted_iv = bits_to_bytes(tmp[0:8*8])
                extracted_length = bits_to_bytes(tmp[8*8:12*8])
                message_length = int.from_bytes(extracted_length)

                if extracted_iv != iv_bytes:
                    print("IV код не совпадает")
                    sys.exit(0)

                used_positions = [0] * width * height * 3
                keystream = []
                tmp = keystreamGen(last_keystream)
                keystream.append(tmp)
                used_positions[tmp % m] = 1
                for i in range(message_length * 8):
                    tmp = keystreamGen(tmp)
                    while used_positions[tmp % m]:
                        tmp = keystreamGen(tmp)
                    keystream.append(tmp)
                    used_positions[tmp % m] = 1
                keystream = correctKeystream(message_length * 8, keystream, width * height * 3)

                tmp = []
                for i in keystream:
                    tmp.append(pixels_flat[i] & 1)

                extracted_message = bits_to_bytes(tmp)

                key_msg = int(args.password.encode('utf-8').hex() + extracted_iv.hex(), 16)
                keystream_msg = keystreamGenLCG(len(extracted_message), key_msg)
                keystream_msg = correctKeystream(len(extracted_message), keystream_msg, 255)

                # xor операция между сообщением и keystream
                ciphertext = xorStream(extracted_message, keystream_msg, len(extracted_message))

                result = ''.join(chr(code) for code in ciphertext)
                with open('../results/decrypted_message.txt', 'w', encoding='utf-8') as file:
                    file.write(result)
            elif args.LSB == "classic":
                tmp = getBits(12*8, pixels, height, width, channels)
                extracted_iv = bits_to_bytes(tmp[0:8*8])
                extracted_length = bits_to_bytes(tmp[8*8:12*8])
                message_length = int.from_bytes(extracted_length)

                if extracted_iv != iv_bytes:
                    print("IV код не совпадает")
                    sys.exit(0)

                tmp = getBits(12 * 8 + message_length * 8, pixels, height, width, channels)

                extractedBits = tmp[12*8:12*8 + message_length*8]

                extracted_message = bits_to_bytes(extractedBits)

                key_msg = int(args.password.encode('utf-8').hex() + extracted_iv.hex(), 16)
                keystream_msg = keystreamGenLCG(len(extracted_message), key_msg)
                keystream_msg = correctKeystream(len(extracted_message), keystream_msg, 255)

                # xor операция между сообщением и keystream
                ciphertext = xorStream(extracted_message, keystream_msg, len(extracted_message))

                result = ''.join(chr(code) for code in ciphertext)
                with open('../results/decrypted_message.txt', 'w', encoding='utf-8') as file:
                    file.write(result)
