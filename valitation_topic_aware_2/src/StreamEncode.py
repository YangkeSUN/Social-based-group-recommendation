
# encode = True for gzip
class StreamEncode:
    stream_out = None
    encode = None

    def __init__(self, stream_out, encode):
        self.stream_out = stream_out
        self.encode = encode
        if (type(encode) == str):
            self.write = self.write_string_encode
        elif (encode == True):
            self.write = self.write_true_encode
        elif (encode == False):
            self.write = self.write_false_encode
        else:
            self.write = self.write_normal
    
    def close(self):
        if (self.stream_out != None):
            self.flush()
            self.stream_out.close()
    
    def write_string_encode(self, string):
        string1 = string.encode(encode)
        self.stream_out.write(string1)
    
    def write_true_encode(self, string):
        string1 = string.encode()
        self.stream_out.write(string1)
    
    def write_false_encode(self, string):
        self.stream_out.write(string)

    def write_normal(self, string):
        self.stream_out.write(string)
    
    def write_flush(self, string):
        self.write(string)
        self.flush()

    def read(self, size=None):
        return self.stream_out.read(size)

    def flush(self):
        self.stream_out.flush()
    
    def read_normal(self, size=None):
        if (size == None):
            return self.stream_out.read()
        return self.stream_out.read(size)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
# End class StreamEncode
