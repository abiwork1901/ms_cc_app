package com.creditcard.actuator;

import com.creditcard.repository.CreditCardRepository;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;

@Component
public class CreditCardServiceHealthIndicator implements HealthIndicator {

    private final CreditCardRepository repository;

    public CreditCardServiceHealthIndicator(CreditCardRepository repository) {
        this.repository = repository;
    }

    @Override
    public Health health() {
        try {
            long count = repository.count();
            return Health.up()
                    .withDetail("totalCards", count)
                    .withDetail("status", "Credit Card Service is healthy")
                    .build();
        } catch (Exception e) {
            return Health.down()
                    .withException(e)
                    .withDetail("status", "Credit Card Service is not healthy")
                    .build();
        }
    }
} 