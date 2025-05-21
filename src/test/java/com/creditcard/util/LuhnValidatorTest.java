package com.creditcard.util;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LuhnValidatorTest {
    @Test
    void validCardNumbers() {
        assertTrue(LuhnValidator.isValid("4539578763621486")); // Visa
        assertTrue(LuhnValidator.isValid("6011000990139424")); // Discover
        assertTrue(LuhnValidator.isValid("378282246310005")); // Amex
    }

    @Test
    void invalidCardNumbers() {
        assertFalse(LuhnValidator.isValid("1234567890123456"));
        assertFalse(LuhnValidator.isValid("1111111111111111"));
        assertFalse(LuhnValidator.isValid(""));
        assertFalse(LuhnValidator.isValid(null));
        assertFalse(LuhnValidator.isValid("abcd1234"));
    }
} 