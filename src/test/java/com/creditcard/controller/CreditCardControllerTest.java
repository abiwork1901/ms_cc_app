package com.creditcard.controller;

import com.creditcard.dto.CreditCardRequest;
import com.creditcard.model.CreditCard;
import com.creditcard.service.CreditCardService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Arrays;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(CreditCardController.class)
public class CreditCardControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private CreditCardService service;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    public void createCard_ValidRequest_ReturnsCreated() throws Exception {
        CreditCardRequest request = new CreditCardRequest("John Doe", "4532015112830366", 1000.0);
        CreditCard response = new CreditCard(1L, "John Doe", "4532015112830366", 1000.0, 0.0);

        when(service.createCard(any(CreditCardRequest.class))).thenReturn(response);

        mockMvc.perform(post("/api/cards")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.name").value("John Doe"))
                .andExpect(jsonPath("$.cardNumber").value("4532015112830366"))
                .andExpect(jsonPath("$.creditLimit").value(1000.0))
                .andExpect(jsonPath("$.balance").value(0.0));
    }

    @Test
    public void createCard_InvalidCardNumber_ReturnsBadRequest() throws Exception {
        CreditCardRequest request = new CreditCardRequest("John Doe", "1234567890123456", 1000.0);

        when(service.createCard(any(CreditCardRequest.class)))
                .thenThrow(new IllegalArgumentException("Invalid card number"));

        mockMvc.perform(post("/api/cards")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.message").value("Invalid card number"));
    }

    @Test
    public void getAllCards_ReturnsListOfCards() throws Exception {
        List<CreditCard> cards = Arrays.asList(
            new CreditCard(1L, "John Doe", "4532015112830366", 1000.0, 0.0),
            new CreditCard(2L, "Jane Doe", "4532015112830367", 2000.0, 0.0)
        );

        when(service.getAllCards()).thenReturn(cards);

        mockMvc.perform(get("/api/cards"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(1))
                .andExpect(jsonPath("$[0].name").value("John Doe"))
                .andExpect(jsonPath("$[1].id").value(2))
                .andExpect(jsonPath("$[1].name").value("Jane Doe"));
    }
} 