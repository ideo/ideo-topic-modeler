
def decode_ascii(x):
  encoded_string = x.encode("ascii", "ignore")
  decode_string = encoded_string.decode()
  return decode_string
