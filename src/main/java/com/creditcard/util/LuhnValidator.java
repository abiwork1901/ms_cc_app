package com.creditcard.util;

public class LuhnValidator {
    public static boolean isValid(String cardNumber) {
        if (cardNumber == null || cardNumber.isEmpty() || !cardNumber.matches("\\d+")) {
            return false;
        }

        int sum = 0;
        boolean alternate = false;
        
        // Loop through values starting from the rightmost digit
        for (int i = cardNumber.length() - 1; i >= 0; i--) {
            int n = Integer.parseInt(cardNumber.substring(i, i + 1));
            if (alternate) {
                n *= 2;
                if (n > 9) {
                    n = (n % 10) + 1;
                }
            }
            sum += n;
            alternate = !alternate;
        }
        
        return (sum % 10 == 0);
    }
} 