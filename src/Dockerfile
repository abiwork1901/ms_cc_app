FROM openjdk:17-slim as build
WORKDIR /workspace/app

COPY pom.xml .
COPY src src

# Install Maven
RUN apt-get update && \
    apt-get install -y maven && \
    mvn package -DskipTests

FROM openjdk:17-slim
VOLUME /tmp
COPY --from=build /workspace/app/target/*.jar app.jar

ENTRYPOINT ["java", "-jar", "/app.jar"] 