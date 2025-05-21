package com.creditcard.actuator;

import com.creditcard.repository.CreditCardRepository;
import org.springframework.boot.actuate.endpoint.annotation.Endpoint;
import org.springframework.boot.actuate.endpoint.annotation.ReadOperation;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
@Endpoint(id = "creditcardstats")
public class CreditCardMetricsEndpoint {

    private final CreditCardRepository repository;

    public CreditCardMetricsEndpoint(CreditCardRepository repository) {
        this.repository = repository;
    }

    @ReadOperation
    public Map<String, Object> getCreditCardStats() {
        Map<String, Object> stats = new HashMap<>();
        
        try {
            long totalCards = repository.count();
            double totalCreditLimit = repository.findAll().stream()
                    .mapToDouble(card -> card.getCreditLimit())
                    .sum();
            double totalBalance = repository.findAll().stream()
                    .mapToDouble(card -> card.getBalance())
                    .sum();
            
            stats.put("totalCards", totalCards);
            stats.put("totalCreditLimit", totalCreditLimit);
            stats.put("totalBalance", totalBalance);
            stats.put("averageCreditLimit", totalCards > 0 ? totalCreditLimit / totalCards : 0);
            stats.put("averageBalance", totalCards > 0 ? totalBalance / totalCards : 0);
            stats.put("status", "success");
        } catch (Exception e) {
            stats.put("status", "error");
            stats.put("error", e.getMessage());
        }
        
        return stats;
    }
} 