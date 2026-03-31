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
