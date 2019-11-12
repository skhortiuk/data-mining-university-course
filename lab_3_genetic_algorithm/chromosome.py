class Chromosome:
    @staticmethod
    def _convert_to_binary(value: int, bin_length: int) -> str:
        actual_len = len(bin(value)[2:])
        if actual_len > bin_length:
            raise ValueError("Binary value greater than allowed `bin_length`")

        return f"{'0' * (bin_length - actual_len)}{bin(value)[2:]}"

    def __init__(self, value: int, bin_length: int):
        self._bit_value = self._convert_to_binary(value, bin_length)
        self._value = value
