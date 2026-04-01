## 🛑 Issue
You deployed an Ingress resource (ai-ingress) but when accessing the external IP (20.253.163.78), you only saw “404 Not Found – nginx”.

The Ingress showed Ingress `Class: <none>` when you described it, meaning it wasn’t bound to the NGINX ingress controller.

## 🔍 Cause
The Ingress manifest was missing the field:

```yaml
spec:
  ingressClassName: nginx
```
Without this, the NGINX ingress controller doesn’t recognize or process the Ingress resource.

As a result, requests to the LoadBalancer IP were handled by the controller’s default backend, which simply returns a 404.

✅ Resolution
Add the ingressClassName: nginx field to the Ingress YAML:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-ingress
  namespace: ai-app
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-service
            port:
              number: 80
```
Reapply the manifest:

```bash
kubectl apply -f ai-ingress.yaml
```
Verify with:

```bash
kubectl describe ingress ai-ingress -n ai-app
```
Now it should show Ingress Class: nginx.

## 📌 What happens if you miss it
The Ingress resource exists but is ignored by the controller.

The external IP is reachable, but traffic never routes to your service/pods.

You only see the default 404 page from NGINX.

So the root issue was missing ingressClassName, and the solution was explicitly binding the Ingress to the NGINX controller.



## ⚠️ Issue Faced
After deploying the AI app with Helm, the Ingress showed an external IP, but accessing http://<EXTERNAL-IP>/docs returned 502 Bad Gateway.

> Root cause:  
The Kubernetes Service was forwarding traffic to targetPort: 8000, but inside the container FastAPI was actually listening on 8081.
This mismatch meant NGINX Ingress could reach the Service, but the Service couldn’t forward traffic correctly to the pods.

## 🛠 Resolution
Identified mismatch by checking Service endpoints and pod configuration.

Service was mapping 80 → 8000.

Pods were listening on 8081.

Updated Helm chart Service template (ai-app-chart/templates/service.yaml):

```yaml
ports:
  - port: 80
    targetPort: 8081
```
Upgraded Helm release to apply the fix:

```bash
helm upgrade ai-release ./ai-app-chart -n ai-app
```
Verified endpoints:

```bash
kubectl get endpoints ai-service -n ai-app
```
Now showed pod IPs bound to 8081.

Tested external access:

```Code
curl http://<EXTERNAL-IP>/docs
```
✅ FastAPI Swagger UI loaded successfully.

✅ Lesson Learned
Always align containerPort in Deployment with targetPort in Service.

> Ingress → Service → Pod chain must be consistent.


