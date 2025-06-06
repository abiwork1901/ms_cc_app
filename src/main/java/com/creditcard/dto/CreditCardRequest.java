package com.creditcard.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CreditCardRequest {
    private String name;
    private String cardNumber;
    private double creditLimit;
} 