package com.creditcard.service;

import com.creditcard.dto.CreditCardRequest;
import com.creditcard.model.CreditCard;
import com.creditcard.repository.CreditCardRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class CreditCardService {
    private final CreditCardRepository repository;
    private static final Logger logger = LoggerFactory.getLogger(CreditCardService.class);

    public CreditCardService(CreditCardRepository repository) {
        this.repository = repository;
    }

    @Transactional
    public CreditCard createCard(CreditCardRequest request) {
        if (!isValidCardNumber(request.getCardNumber())) {
            String errorMessage = String.format("Invalid card number provided: ****-****-****-%s for user: %s",
                    request.getCardNumber().substring(request.getCardNumber().length() - 4),
                    request.getName());
            logger.error(errorMessage);
            throw new IllegalArgumentException(errorMessage);
        }

        CreditCard card = new CreditCard();
        card.setName(request.getName());
        card.setCardNumber(request.getCardNumber());
        card.setCreditLimit(request.getCreditLimit());
        card.setBalance(0.0);

        return repository.save(card);
    }

    public List<CreditCard> getAllCards() {
        return repository.findAll();
    }

    private boolean isValidCardNumber(String cardNumber) {
        if (cardNumber == null || cardNumber.length() < 13 || cardNumber.length() > 19) {
            return false;
        }

        int sum = 0;
        boolean alternate = false;

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