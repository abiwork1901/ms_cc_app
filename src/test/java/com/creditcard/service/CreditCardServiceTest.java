package com.creditcard.service;

import com.creditcard.dto.CreditCardRequest;
import com.creditcard.model.CreditCard;
import com.creditcard.repository.CreditCardRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class CreditCardServiceTest {

    @Mock
    private CreditCardRepository repository;

    @InjectMocks
    private CreditCardService service;

    @Test
    public void createCard_ValidCardNumber_ReturnsSavedCard() {
        // Arrange
        CreditCardRequest request = new CreditCardRequest("John Doe", "4532015112830366", 1000.0);
        CreditCard expectedCard = new CreditCard(1L, "John Doe", "4532015112830366", 1000.0, 0.0);
        when(repository.save(any(CreditCard.class))).thenReturn(expectedCard);

        // Act
        CreditCard result = service.createCard(request);

        // Assert
        assertNotNull(result);
        assertEquals(expectedCard.getId(), result.getId());
        assertEquals(expectedCard.getName(), result.getName());
        assertEquals(expectedCard.getCardNumber(), result.getCardNumber());
        assertEquals(expectedCard.getCreditLimit(), result.getCreditLimit());
        assertEquals(expectedCard.getBalance(), result.getBalance());
    }

    @Test
    public void createCard_InvalidCardNumber_ThrowsException() {
        // Arrange
        CreditCardRequest request = new CreditCardRequest("John Doe", "1234567890123456", 1000.0);

        // Act & Assert
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> service.createCard(request));
        assertTrue(exception.getMessage().contains("Invalid card number"));
    }

    @Test
    public void getAllCards_ReturnsListOfCards() {
        // Arrange
        List<CreditCard> expectedCards = Arrays.asList(
            new CreditCard(1L, "John Doe", "4532015112830366", 1000.0, 0.0),
            new CreditCard(2L, "Jane Doe", "4532015112830367", 2000.0, 0.0)
        );
        when(repository.findAll()).thenReturn(expectedCards);

        // Act
        List<CreditCard> result = service.getAllCards();

        // Assert
        assertNotNull(result);
        assertEquals(2, result.size());
        assertEquals(expectedCards.get(0).getName(), result.get(0).getName());
        assertEquals(expectedCards.get(1).getName(), result.get(1).getName());
    }
} 