package com.creditcard.controller;

import com.creditcard.dto.CreditCardRequest;
import com.creditcard.model.CreditCard;
import com.creditcard.service.CreditCardService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/cards")
@CrossOrigin(origins = "*")
@Tag(name = "Credit Card Controller", description = "APIs for managing credit cards")
public class CreditCardController {
    private final CreditCardService service;
    private static final Logger logger = LoggerFactory.getLogger(CreditCardController.class);

    public CreditCardController(CreditCardService service) {
        this.service = service;
    }

    @Operation(summary = "Create a new credit card", description = "Creates a new credit card with the provided details")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "201", description = "Credit card created successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid input data")
    })
    @PostMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> createCard(
            @Parameter(description = "Credit card details", required = true)
            @RequestBody CreditCardRequest request) {
        try {
            CreditCard card = service.createCard(request);
            return ResponseEntity.status(HttpStatus.CREATED).body(card);
        } catch (IllegalArgumentException e) {
            String errorMessage = e.getMessage();
            logger.error("Error creating card: {}", errorMessage);
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Map.of("message", errorMessage));
        }
    }

    @Operation(summary = "Get all credit cards", description = "Retrieves a list of all credit cards")
    @ApiResponse(responseCode = "200", description = "Successfully retrieved all credit cards")
    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<CreditCard>> getAllCards() {
        return ResponseEntity.ok(service.getAllCards());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ResponseEntity<Map<String, String>> handleIllegalArgumentException(IllegalArgumentException e) {
        String errorMessage = e.getMessage();
        logger.error("Error handling request: {}", errorMessage);
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of("message", errorMessage));
    }
} 