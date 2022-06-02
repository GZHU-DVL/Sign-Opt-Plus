import time
import torch
from utils import Logger

class Sign_OPT_Plus_l2(object):
    def __init__(self, model, k=200, initial_dataset=None, log_path=None, targ_flag=None):
        self.model = model
        self.k = k
        self.initial_dataset = initial_dataset
        self.logger = Logger(log_path)
        self.targ_flag = targ_flag

    def attack_untargeted(self, x0, y0, eps, eta=0.2, alpha=1, beta=0.75, iterations=20000, query_limit=20000, h=0.001):
        """ Attack the original image and return adversarial example
            paper_model: (pytorch paper_model)
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0
        suc_count, fail_count, unclean_count = 0, 0, 0

        ## check clean
        if (model.predict_label(x0) != y0):
            self.logger.log("Fail to classify the image. No need to attack.")
            unclean_count += 1
            return x0, 0, suc_count, fail_count, unclean_count

        #######################Begin Attack ##########################
        #### S1: Init: Calculate a good starting point (direction)
        ini_iter = 100
        best_theta, g_theta = None, float('inf')
        self.logger.log("Searching for the initial direction on {} random directions: " .format(ini_iter))
        timestart = time.time()
        for i in range(ini_iter):
            query_count += 1
            theta = torch.randn(*x0.shape).cuda()  # gaussian distortion
            # register adv directions
            if model.predict_label(x0 + theta) != y0:
                theta_norm = torch.norm(theta)
                theta /= theta_norm
                lbd, count = self.binary_search(model, x0, y0, theta, theta_norm.cpu().numpy(), g_theta, tol=1e-3)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.logger.log("--------> Found distortion {:.4f}".format(g_theta))
        timeend = time.time()
        ## fail if cannot find a adv direction
        if g_theta == float('inf'):
            self.logger.log("Couldn't find valid initial, fail_counted")
            fail_count += 1
            return x0, 0, suc_count, fail_count, unclean_count
        self.logger.log("==========> Found best distortion {:.4f} in {:.4f} seconds "
              "using {} queries".format(g_theta, timeend - timestart, query_count))

        #### S2: Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        for i in range(iterations):
            sign_gradient, grad_queries = self.grad_est(x0, y0, xg, initial_lbd=gg, h=h)
            ## Line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg  ## next theta
            min_g2 = gg  ## current g_theta
              ## velocity (for momentum only)
            for _ in range(15):
                # update theta by one step sgd
                new_theta = xg - eta * sign_gradient
                new_theta /= torch.norm(new_theta)
                ls_count += 1
                if model.predict_label(x0 + min_g2 * new_theta) != y0:
                    new_g2, count = self.binary_search_local(
                        model, x0, y0, new_theta, min_g2, tol=h/500)
                else:
                    new_g2, count = float('inf'), 0
                ls_count += count
                eta = (1 + alpha) * eta  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:  ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    eta = (1 - beta) * eta
                    new_theta = xg - eta * sign_gradient
                    new_theta /= torch.norm(new_theta)
                    ls_count += 1
                    if model.predict_label(x0 + min_g2 * new_theta) != y0:
                        new_g2, count = self.binary_search_local(
                            model, x0, y0, new_theta, min_g2, tol=h/500)
                    else:
                        new_g2, count = float('inf'), 0
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if eta < 1e-4:  ## if the above two blocks of code failed
                eta = 1.0
                self.logger.log("Warning: not moving")
                h = h * 0.1
                if (h < 1e-8):
                    break

            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            self.logger.log('distortion:{:.4f}'.format(gg))

            ## logging
            if (i + 1) % 10 == 0:
                self.logger.log("Iteration %3d distortion %.4f num_queries %d" % (i + 1, gg, query_count))

            if (query_count > query_limit):
                x_adv = x0 + gg * xg.clone().detach().cuda()
                x_adv = x_adv.clamp(0,1)
                dis = torch.norm(x_adv - x0)
                if(gg < eps):
                    suc_count += 1
                    return x_adv, dis, suc_count, fail_count, unclean_count
                else:
                    fail_count += 1
                    return x_adv, dis, suc_count, fail_count, unclean_count

        x_adv = x0 + torch.tensor(gg * xg, dtype=torch.float).cuda()
        x_adv = x_adv.clamp(0, 1)
        dis = torch.norm(x_adv - x0)
        if gg < eps:
            suc_count += 1
            target = model.predict_label(x_adv)
            self.logger.log("Succeed distortion {:.4f} target"
                  " {:d} queries {:d} LS queries {:d}\n".format(gg, target, query_count, ls_total))
            return x_adv, dis, suc_count, fail_count, unclean_count
        else:
            fail_count += 1
            self.logger.log("\nFailed: distortion {:.4f}".format(gg))
            return x_adv, dis, suc_count, fail_count, unclean_count

    def attack_targeted(self, x0, y0, eps, eta=0.2, alpha=3, beta=0.75, iterations=20000, query_limit=40000, h=0.001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x0, y0): original image
        """

        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0
        suc_count, fail_count, unclean_count = 0, 0, 0
        target = (y0 + 1) % model.num_classes
        self.logger.log("Targeted attack - Source: {0} and Target: {1}".format(y0, target.item()))

        if self.initial_dataset is None:
            self.logger.log("Need training dataset for initial theta.")
            return x0, 0, suc_count, fail_count, unclean_count
        if (model.predict_label(x0) != y0):
            self.logger.log("Fail to classify the image. No need to attack.")
            unclean_count += 1
            return x0, 0, suc_count, fail_count, unclean_count

        ### Initial start point
        if model.dataset == 'MNIST' or model.dataset == 'CIFAR10':
            alpha = 1
            num_samples = 100
            best_theta, g_theta = None, float('inf')
            self.logger.log("Searching for the initial direction on %d samples: " % (num_samples))
            timestart = time.time()
            sample_count = 0
            for i, (xi, yi) in enumerate(self.initial_dataset):
                xi = xi.cuda()
                yi_pred = model.predict_label(xi)
                query_count += 1
                if yi_pred != target:
                    continue
                theta = xi - x0
                theta_norm = torch.norm(theta)
                theta /= theta_norm
                lbd, count = self.binary_search(model, x0, target, theta, theta_norm.cpu().numpy(), g_theta, tol=1e-5)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.logger.log("--------> Found distortion %.4f" % g_theta)
                sample_count += 1
                if sample_count >= num_samples:
                    break
                if i > 500:
                    break

        elif model.dataset == 'ImageNet':
            alpha = 3
            ini_iter = 50
            best_theta, g_theta = None, float('inf')
            self.logger.log("Searching for the initial direction on %d samples: " % (ini_iter))
            timestart = time.time()
            for i in range(ini_iter):
                xi = self.initial_dataset[target * ini_iter + i][0]
                xi = xi.cuda()
                yi_pred = model.predict_label(xi)
                query_count += 1
                if yi_pred != target:
                    continue
                theta = xi - x0
                theta_norm = torch.norm(theta)
                theta /= theta_norm
                lbd, count = self.binary_search(model, x0, target, theta, theta_norm.cpu().numpy(), g_theta, tol=1e-5)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.logger.log("--------> Found distortion {:.4f}".format(g_theta))

        timeend = time.time()
        if g_theta == float('inf'):
            self.logger.log("Couldn't find valid initial, failed")
            fail_count += 1
            return x0, 0, suc_count, fail_count, unclean_count
        self.logger.log("==========> Found best distortion {:.4f} in {:.4f} seconds "
                        "using {} queries".format(g_theta, timeend - timestart, query_count))

        #### S2: Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        for i in range(iterations):
            sign_gradient, grad_queries = self.grad_est(x0, target, xg, initial_lbd=gg, h=h)
            ## Line search of the step size of gradient descent
            ls_count = 0
            min_theta = xg  ## next theta
            min_g2 = gg  ## current g_theta

            for _ in range(15):
                # update theta by one step sgd
                new_theta = xg - eta * sign_gradient
                new_theta /= torch.norm(new_theta)
                ls_count += 1
                if model.predict_label(x0 + min_g2 * new_theta) == target:
                    new_g2, count = self.binary_search_local(
                        model, x0, target, new_theta, min_g2, tol=h/500)
                else:
                    new_g2, count = float('inf'), 0
                ls_count += count
                eta = (1 + alpha) * eta  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:  ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    eta = (1 - beta) * eta
                    new_theta = xg - eta * sign_gradient
                    new_theta /= torch.norm(new_theta)
                    ls_count += 1
                    if model.predict_label(x0 + min_g2 * new_theta) == target:
                        new_g2, count = self.binary_search_local(
                            model, x0, target, new_theta, min_g2, tol=h/500)
                    else:
                        new_g2, count = float('inf'), 0
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if eta < 1e-4:  ## if the above two blocks of code failed
                eta = 1.0
                self.logger.log("Warning: not moving")
                h = h * 0.1
                if (h < 1e-8):
                    break

            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            self.logger.log('distortion:{:.4f}'.format(gg))

            if query_count > query_limit:
                break

            if (i + 1) % 10 == 0:
                self.logger.log("Iteration %3d distortion %.4f num_queries %d" % (i + 1, gg, query_count))

        adv_target = model.predict_label(x0 + gg * xg)
        x_adv = x0 + gg * xg.clone().detach().cuda()
        x_adv = x_adv.clamp(0, 1)
        dis = torch.norm(x_adv - x0)
        if (adv_target == target):
            timeend = time.time()
            self.logger.log("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d LS queries %d \nTime: %.4f seconds" % (
                      gg, target, query_count, ls_total, timeend - timestart))
            if (gg < eps):
                suc_count += 1
                return x_adv, dis, suc_count, fail_count, unclean_count
            else:
                fail_count += 1
                return x_adv, dis, suc_count, fail_count, unclean_count
        else:
            fail_count += 1
            self.logger.log("Failed to find targeted adversarial example.")
            return x_adv, dis, suc_count, fail_count, unclean_count

    def grad_est(self, x0, y0, theta, initial_lbd, h=0.001):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k  # 200 random directions (for estimating the gradient)
        sign_grad = torch.zeros(theta.size()).cuda()
        queries = 0
        h = 0.001
        for _ in range(K):  # for each u
            u = torch.randn(theta.size()).cuda()
            u /= torch.norm(u)
            new_theta = (theta + h * u).cuda()
            new_theta /= torch.norm(new_theta)
            sign = 1

            # Targeted case.
            if (self.targ_flag and self.model.predict_label(x0 + initial_lbd * new_theta) == y0):
                sign = -1

            # Untargeted case
            if not self.targ_flag and self.model.predict_label(x0 + initial_lbd * new_theta) != y0:  # success
                sign = -1

            queries += 1

            sign_grad += u * sign

        sign_grad /= K

        return sign_grad, queries

    def binary_search_local(self, model, x0, y0, theta, initial_lbd, tol=2e-6):
        nquery = 0
        lbd = initial_lbd
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        ## Untargeted
        if not self.targ_flag:
            while model.predict_label(x0 + lbd_lo * theta) != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

            while (lbd_hi - lbd_lo) > tol:
                lbd_mid = (lbd_lo + lbd_hi) / 2.0
                nquery += 1
                if model.predict_label(x0 + lbd_mid * theta) != y0:
                    lbd_hi = lbd_mid
                else:
                    lbd_lo = lbd_mid
        ## Targeted
        else:
            while model.predict_label(x0 + lbd_lo * theta) == y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

            while (lbd_hi - lbd_lo) > tol:
                lbd_mid = (lbd_lo + lbd_hi) / 2.0
                nquery += 1
                if model.predict_label(x0 + lbd_mid * theta) == y0:
                    lbd_hi = lbd_mid
                else:
                    lbd_lo = lbd_mid
        return lbd_hi, nquery

    def binary_search(self, model, x0, y0, theta, initial_lbd, current_best, tol=1e-3):
        nquery = 0
        if initial_lbd > current_best:
            ## Untargeted
            if not self.targ_flag and model.predict_label(x0 + current_best * theta) == y0:
                nquery += 1
                return float('inf'), nquery
            ## Targeted
            if self.targ_flag and model.predict_label(x0 + current_best * theta) != y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            ## Untargeted
            if not self.targ_flag:
                if model.predict_label(x0 + lbd_mid * theta) != y0:
                    lbd_hi = lbd_mid
                else:
                    lbd_lo = lbd_mid
            ## Targeted
            else:
                if model.predict_label(x0 + lbd_mid * theta) == y0:
                    lbd_hi = lbd_mid
                else:
                    lbd_lo = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, distortion=None, query_limit=4000):
        if self.targ_flag:
            adv = self.attack_targeted(input_xi, label_or_target, eps=distortion, query_limit=query_limit)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, eps=distortion, query_limit=query_limit)
        return adv